#!/usr/bin/env python3
"""
Database Connection and Tool Testing Script

Run this to verify your database is accessible and tools work correctly.

Usage:
    python test_database.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_database_connection():
    """Test basic database connectivity."""
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    
    try:
        from sqlalchemy import create_engine
        from langchain_community.utilities import SQLDatabase
        from pathlib import Path
        
        # Adjust this path to match your setup
        BASE_DIR = Path(__file__).resolve().parents[1]
        DB_NAME = "new_sb3.db"  # Change this to your actual DB name
        DB_PATH = BASE_DIR / DB_NAME
        
        print(f"Database path: {DB_PATH}")
        print(f"Database exists: {DB_PATH.exists()}")
        
        if not DB_PATH.exists():
            print("❌ Database file not found!")
            print(f"Please check the path: {DB_PATH}")
            return False
        
        # Create engine and connect
        engine = create_engine(f"sqlite:///{DB_PATH}")
        sql_db = SQLDatabase(engine=engine)
        
        # Test getting tables
        tables = sql_db.get_usable_table_names()
        print(f"✅ Connected successfully!")
        print(f"✅ Found {len(tables)} tables: {tables}")
        
        # Test getting schema for first table
        if tables:
            table_info = sql_db.get_table_info([tables[0]])
            print(f"\n✅ Schema for '{tables[0]}':")
            print(table_info[:500] + "..." if len(table_info) > 500 else table_info)
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sql_tools():
    """Test that SQL tools can be created and used."""
    print("\n" + "="*60)
    print("TEST 2: SQL Tools Creation")
    print("="*60)
    
    try:
        from sqlalchemy import create_engine
        from langchain_community.utilities import SQLDatabase
        from pathlib import Path
        from sb3_api.repository.agent.sql_generic import SQLDatabaseWrapper
        from sb3_api.agent.tools.sql import (
            ListRedshiftSQLDatabaseTool,
            InfoRedshiftSQLDatabaseTool,
            QueryRedshiftSQLDatabaseTool,
        )
        
        # Setup database
        BASE_DIR = Path(__file__).resolve().parents[1]
        DB_NAME = "new_sb3.db"
        DB_PATH = BASE_DIR / DB_NAME
        engine = create_engine(f"sqlite:///{DB_PATH}")
        sql_db = SQLDatabase(engine=engine)
        db_wrapper = SQLDatabaseWrapper(sql_db)
        
        # Test 1: ListRedshiftSQLDatabaseTool
        print("\n--- Testing ListRedshiftSQLDatabaseTool ---")
        list_tool = ListRedshiftSQLDatabaseTool(db=db_wrapper)
        print(f"Tool name: {list_tool.name}")
        print(f"Has args_schema: {list_tool.args_schema is not None}")
        
        result = list_tool.invoke(input="")
        print(f"✅ List tables result: {result}")
        
        # Test 2: InfoRedshiftSQLDatabaseTool
        print("\n--- Testing InfoRedshiftSQLDatabaseTool ---")
        info_tool = InfoRedshiftSQLDatabaseTool(db=db_wrapper)
        print(f"Tool name: {info_tool.name}")
        print(f"Has args_schema: {info_tool.args_schema is not None}")
        
        # Get first table name
        tables = db_wrapper.get_usable_table_names()
        if tables:
            result = info_tool.invoke(input=tables[0])
            print(f"✅ Get schema result: {result[:200]}...")
        
        # Test 3: QueryRedshiftSQLDatabaseTool
        print("\n--- Testing QueryRedshiftSQLDatabaseTool ---")
        query_tool = QueryRedshiftSQLDatabaseTool(db=db_wrapper)
        print(f"Tool name: {query_tool.name}")
        print(f"Has args_schema: {query_tool.args_schema is not None}")
        
        if tables:
            test_query = {
                "query": f"SELECT * FROM {tables[0]} LIMIT 1",
                "query_purpose": "clarification"
            }
            result = query_tool.invoke(input=test_query)
            print(f"✅ Query result: {result[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_wrapping():
    """Test that tools work with Strands wrapping."""
    print("\n" + "="*60)
    print("TEST 3: Tool Wrapping for Strands")
    print("="*60)
    
    try:
        from sqlalchemy import create_engine
        from langchain_community.utilities import SQLDatabase
        from pathlib import Path
        from sb3_api.repository.agent.sql_generic import SQLDatabaseWrapper
        from sb3_api.agent.tools.sql import ListRedshiftSQLDatabaseTool
        from strands import tool as strands_tool
        
        # Setup database
        BASE_DIR = Path(__file__).resolve().parents[1]
        DB_NAME = "new_sb3.db"
        DB_PATH = BASE_DIR / DB_NAME
        engine = create_engine(f"sqlite:///{DB_PATH}")
        sql_db = SQLDatabase(engine=engine)
        db_wrapper = SQLDatabaseWrapper(sql_db)
        
        # Create a LangChain tool
        lc_tool = ListRedshiftSQLDatabaseTool(db=db_wrapper)
        
        # Wrap it for Strands
        input_schema = {
            "json": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input parameter"
                    }
                },
                "required": []
            }
        }
        
        def _wrapper(**kwargs):
            tool_input = kwargs.get("input", "")
            result = lc_tool.invoke(input=tool_input)
            return {"status": "success", "content": [{"text": str(result)}]}
        
        wrapped_tool = strands_tool(
            name="sql_db_list_tables",
            description=lc_tool.description,
            inputSchema=input_schema,
        )(_wrapper)
        
        # Test calling the wrapped tool
        result = wrapped_tool(input="")
        print(f"✅ Wrapped tool result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool wrapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_query():
    """Test a specific query like 'What are available TV products'."""
    print("\n" + "="*60)
    print("TEST 4: Specific TV Products Query")
    print("="*60)
    
    try:
        from sqlalchemy import create_engine
        from langchain_community.utilities import SQLDatabase
        from pathlib import Path
        from sb3_api.repository.agent.sql_generic import SQLDatabaseWrapper
        
        # Setup database
        BASE_DIR = Path(__file__).resolve().parents[1]
        DB_NAME = "new_sb3.db"
        DB_PATH = BASE_DIR / DB_NAME
        engine = create_engine(f"sqlite:///{DB_PATH}")
        sql_db = SQLDatabase(engine=engine)
        db_wrapper = SQLDatabaseWrapper(sql_db)
        
        # Get all tables
        tables = db_wrapper.get_usable_table_names()
        print(f"Available tables: {tables}")
        
        # Look for a products table
        products_tables = [t for t in tables if 'product' in t.lower()]
        
        if products_tables:
            table_name = products_tables[0]
            print(f"\nFound products table: {table_name}")
            
            # Get schema
            schema = db_wrapper.get_table_info([table_name])
            print(f"\nTable schema:\n{schema}")
            
            # Try to query for TV products
            test_queries = [
                f"SELECT * FROM {table_name} WHERE product_name LIKE '%TV%' LIMIT 5",
                f"SELECT * FROM {table_name} WHERE LOWER(product_name) LIKE '%tv%' LIMIT 5",
                f"SELECT * FROM {table_name} LIMIT 5",
            f"SELECT strftime('%Y-%m-01', order_date) AS month,COUNT(*) AS order_count FROM src_exp_sales_dm_sb3 WHERE product_type = 'TV' AND strftime('%Y', order_date) = '2024' GROUP BY strftime('%Y-%m-01', order_date) ORDER BY month;"
            ]
            
            for query in test_queries:
                try:
                    print(f"\nTrying query: {query}")
                    result = db_wrapper.run_no_throw(query)
                    print(f"✅ Result: {result}")
                    break
                except Exception as e:
                    print(f"Query failed: {e}")
                    continue
        else:
            print("❌ No products table found")
            print("Please check your database schema")
        
        return True
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DATABASE AND TOOL TESTING SCRIPT")
    print("="*60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("SQL Tools Creation", test_sql_tools),
        ("Tool Wrapping", test_tool_wrapping),
        ("Specific Query", test_specific_query),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Your setup is working!")
    else:
        print("❌ SOME TESTS FAILED - Please review errors above")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
